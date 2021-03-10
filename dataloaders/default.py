import os

import torch
import torch.utils.data as data
import torch.utils.data.ConcatDataset as ConcatDataset
import torchvision
from torchvision import transforms
from .sampler import RandSubClassSampler
from PIL import Image
import numpy as np


def MNIST(batch_sz, num_workers=2):
    normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))

    train_dataset = torchvision.datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=num_workers)
    train_loader.num_classes = 10

    eval_dataset = torchvision.datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize
    ]))
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_sz, shuffle=False, num_workers=num_workers)
    eval_loader.num_classes = 10

    return train_loader, eval_loader

def CIFAR10(batch_sz, num_workers=2):
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

    train_dataset = torchvision.datasets.CIFAR10(
        root='data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=num_workers)
    train_loader.num_classes = 10

    test_dataset = torchvision.datasets.CIFAR10(
        root='data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_sz, shuffle=False, num_workers=num_workers)
    eval_loader.num_classes = 10

    return train_loader, eval_loader


def CIFAR100(batch_sz, num_workers=2):
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

    train_dataset = torchvision.datasets.CIFAR100(
        root='data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=num_workers)
    train_loader.num_classes = 100

    test_dataset = torchvision.datasets.CIFAR100(
        root='data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_sz, shuffle=False, num_workers=num_workers)
    eval_loader.num_classes = 100

    return train_loader, eval_loader

def OfficeHome(batch_sz, num_workers=1, root_dir='data/', source_name='art', target_name='clipart', num_instances=4,
              dataloading='random', strong_augmentation=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dset_transforms = {
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        'strong_train_transforms': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.6, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
        'train_transform': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
        # no data augmentations
        'test_transform': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    }
    
    if dataloading == 'random':
        train_datasets = list()
        for src_domain in source_name:
            train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(root_dir, src_domain),
                            transform=dset_transforms['strong_train_transforms'] if strong_augmentation else dset_transforms['train_transform'])
            train_datasets.append(train_dataset)
        train_dataset = ConcatDataset(train_datasets)
    elif dataloading == 'balanced':
        train_dataset = OfficeHomeBalancedDataset(root_dir=root_dir, source_name=source_name, num_classes=65, 
                            transform=dset_transforms['strong_train_transforms'] if strong_augmentation else dset_transforms['train_transform'], 
                            num_instances=num_instances)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=num_workers)
    train_loader.num_classes = 65

    test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(root_dir, target_name),
                                                     transform=dset_transforms['test_transform'])
    
    eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_sz * num_instances if dataloading == 'balanced' else batch_sz, 
                                        shuffle=False, num_workers=num_workers)
    eval_loader.num_classes = 65

    return train_loader, eval_loader

class OfficeHomeBalancedDataset(data.Dataset):
    def __init__(self, root_dir, source_name, num_classes, transform, num_instances=4):
        super(OfficeHomeBalancedDataset, self).__init__()
        self.root_dir = root_dir
        self.source_name = source_name
        self.num_instances = num_instances
        self.num_classes = num_classes
        self.source_images = self.load_dataset() # create a dict of lists
        self.transform = transform
    
    def load_dataset(self):
        source_image_list = {key: [] for key in range(self.num_classes)}
        self.num_images = 0
        # iterate over all the source domains
        for src_domain in self.source_name:
            source_path = os.path.join(self.root_dir, src_domain + '.txt')
            with open(source_path) as f:
                for ind, line in enumerate(f.readlines()):
                    self.num_images += 1
                    image_dir, label = line.split(' ')
                    img_path = os.path.join(self.root_dir, src_domain, image_dir)
                    source_image_list[int(label)].append(img_path)
        
        return source_image_list

    def __len__(self):
        return int(self.num_images / self.num_instances)

    def __getitem__(self, item):
        image_data = []
        label_data = []

        #np.random.seed(item)
        sampled_class = np.random.choice(range(self.num_classes), size=1, replace=False)[0] # pick a class at random
        t = self.source_images[sampled_class] # get all the instance paths of the sampled class
        if len(t) >= self.num_instances:
            sample_idxs = np.random.choice(range(len(t)), size=self.num_instances, replace=False)
        else:
            sample_idxs = np.random.choice(range(len(t)), size=self.num_instances, replace=True)
        
        # apply image transforms
        for i, sample_idx in enumerate(sample_idxs):
            img_path = self.source_images[sampled_class][sample_idx]
            img = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            image_data.append(img)
        label_data.append([sampled_class] * self.num_instances)

        image_data = torch.stack(image_data)
        label_data = torch.LongTensor(label_data)

        return image_data, label_data


def Omniglot(batch_sz, num_workers=2):
    # This dataset is only for training the Similarity Prediction Network on Omniglot background set
    binary_flip = transforms.Lambda(lambda x: 1 - x)
    normalize = transforms.Normalize((0.086,), (0.235,))
    train_dataset = torchvision.datasets.Omniglot(
        root='data', download=True, background=True,
        transform=transforms.Compose(
           [transforms.RandomResizedCrop(32, (0.85, 1.)),
            transforms.ToTensor(),
            binary_flip,
            normalize]
        ))
    train_length = len(train_dataset)
    train_imgid2cid = [train_dataset[i][1] for i in range(train_length)]  # train_dataset[i] returns (img, cid)
    # Randomly select 20 characters from 964. By default setting (batch_sz=100), each character has 5 images in a mini-batch.
    train_sampler = RandSubClassSampler(
        inds=range(train_length),
        labels=train_imgid2cid,
        cls_per_batch=20,
        batch_size=batch_sz,
        num_batch=train_length//batch_sz)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_sz, shuffle=False,
                                               num_workers=num_workers, sampler=train_sampler)
    train_loader.num_classes = 964

    test_dataset = torchvision.datasets.Omniglot(
        root='data', download=True, background=False,
        transform=transforms.Compose(
          [transforms.Resize(32),
           transforms.ToTensor(),
           binary_flip,
           normalize]
        ))
    eval_length = len(test_dataset)
    eval_imgid2cid = [test_dataset[i][1] for i in range(eval_length)]
    eval_sampler = RandSubClassSampler(
        inds=range(eval_length),
        labels=eval_imgid2cid,
        cls_per_batch=20,
        batch_size=batch_sz,
        num_batch=eval_length // batch_sz)
    eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_sz, shuffle=False,
                                              num_workers=num_workers, sampler=eval_sampler)
    eval_loader.num_classes = 659

    return train_loader, eval_loader


def omniglot_alphabet_func(alphabet, background):
    def create_alphabet_dataset(batch_sz, num_workers=2):
        # This dataset is only for unsupervised clustering
        # train_dataset (with data augmentation) is used during the optimization of clustering criteria
        # test_dataset (without data augmentation) is used after the clustering is converged

        binary_flip = transforms.Lambda(lambda x: 1 - x)
        normalize = transforms.Normalize((0.086,), (0.235,))

        train_dataset = torchvision.datasets.Omniglot(
            root='data', download=True, background=background,
            transform=transforms.Compose(
               [transforms.RandomResizedCrop(32, (0.85, 1.)),
                transforms.ToTensor(),
                binary_flip,
                normalize]
            ))

        # Following part dependents on the internal implementation of official Omniglot dataset loader
        # Only use the images which has alphabet-name in their path name (_characters[cid])
        valid_flat_character_images = [(imgname,cid) for imgname,cid in train_dataset._flat_character_images if alphabet in train_dataset._characters[cid]]
        ndata = len(valid_flat_character_images)  # The number of data after filtering
        train_imgid2cid = [valid_flat_character_images[i][1] for i in range(ndata)]  # The tuple (valid_flat_character_images[i]) are (img, cid)
        cid_set = set(train_imgid2cid)  # The labels are not 0..c-1 here.
        cid2ncid = {cid:ncid for ncid,cid in enumerate(cid_set)}  # Create the mapping table for New cid (ncid)
        valid_characters = {cid2ncid[cid]:train_dataset._characters[cid] for cid in cid_set}
        for i in range(ndata):  # Convert the labels to make sure it has the value {0..c-1}
            valid_flat_character_images[i] = (valid_flat_character_images[i][0],cid2ncid[valid_flat_character_images[i][1]])

        # Apply surgery to the dataset
        train_dataset._flat_character_images = valid_flat_character_images
        train_dataset._characters = valid_characters

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_sz, shuffle=True,
                                                   num_workers=num_workers)
        train_loader.num_classes = len(cid_set)

        test_dataset = torchvision.datasets.Omniglot(
            root='data', download=True, background=background,
            transform=transforms.Compose(
              [transforms.Resize(32),
               transforms.ToTensor(),
               binary_flip,
               normalize]
            ))

        # Apply surgery to the dataset
        test_dataset._flat_character_images = valid_flat_character_images  # Set the new list to the dataset
        test_dataset._characters = valid_characters

        eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_sz, shuffle=False,
                                                  num_workers=num_workers)
        eval_loader.num_classes = train_loader.num_classes

        print('=> Alphabet %s has %d characters and %d images.'%(alphabet, train_loader.num_classes, len(train_dataset)))
        return train_loader, eval_loader
    return create_alphabet_dataset

omniglot_evaluation_alphabets_mapping = {
    'Malayalam':'Malayalam',
     'Kannada':'Kannada',
     'Syriac':'Syriac_(Serto)',
     'Atemayar_Qelisayer':'Atemayar_Qelisayer',
     'Gurmukhi':'Gurmukhi',
     'Old_Church_Slavonic':'Old_Church_Slavonic_(Cyrillic)',
     'Manipuri':'Manipuri',
     'Atlantean':'Atlantean',
     'Sylheti':'Sylheti',
     'Mongolian':'Mongolian',
     'Aurek':'Aurek-Besh',
     'Angelic':'Angelic',
     'ULOG':'ULOG',
     'Oriya':'Oriya',
     'Avesta':'Avesta',
     'Tibetan':'Tibetan',
     'Tengwar':'Tengwar',
     'Keble':'Keble',
     'Ge_ez':'Ge_ez',
     'Glagolitic':'Glagolitic'
}

# Create the functions to access the individual alphabet dataset in Omniglot
for funcName, alphabetStr in omniglot_evaluation_alphabets_mapping.items():
    locals()['Omniglot_eval_' + funcName] = omniglot_alphabet_func(alphabet=alphabetStr, background=False)
